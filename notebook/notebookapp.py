"""A tornado based Jupyter notebook server."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import notebook
import asyncio
import binascii
import datetime
import errno
import functools
import gettext
import hashlib
import hmac
import importlib
import inspect
import io
import ipaddress
import json
import logging
import mimetypes
import os
import random
import re
import select
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
import warnings
import webbrowser

try:
    import resource
except ImportError:
    # Windows
    resource = None

from base64 import encodebytes

from jinja2 import Environment, FileSystemLoader

from jupyter_server.transutils import trans, _

# check for tornado 3.1.0
try:
    import tornado
except ImportError as e:
    raise ImportError(_("The Jupyter Notebook requires tornado >= 5.0")) from e
try:
    version_info = tornado.version_info
except AttributeError as e:
    raise ImportError(_("The Jupyter Notebook requires tornado >= 5.0, but you have < 1.1.0")) from e
if version_info < (5,0):
    raise ImportError(_("The Jupyter Notebook requires tornado >= 5.0, but you have %s") % tornado.version)

from tornado import httpserver
from tornado import ioloop
from tornado import web
from tornado.httputil import url_concat
from tornado.log import LogFormatter, app_log, access_log, gen_log
if not sys.platform.startswith('win'):
    from tornado.netutil import bind_unix_socket

from notebook import (
    DEFAULT_NOTEBOOK_PORT,
    DEFAULT_STATIC_FILES_PATH,
    DEFAULT_TEMPLATE_PATH_LIST,
    __version__,
)

from jupyter_server.extension.application import ExtensionApp
from jupyter_server.extension.application import ExtensionAppJinjaMixin


from .log import log_request


from traitlets.config import Config
from traitlets.config.application import catch_config_error, boolean_flag
from jupyter_core.application import (
    JupyterApp, base_flags, base_aliases,
)
from jupyter_core.paths import jupyter_config_path
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.session import Session
from nbformat.sign import NotebookNotary
from traitlets import (
    Any, Dict, Unicode, Integer, List, Bool, Bytes, Instance,
    TraitError, Type, Float, observe, default, validate
)
from ipython_genutils import py3compat
from jupyter_core.paths import jupyter_runtime_dir, jupyter_path
from jupyter_server._sysinfo import get_sys_info

from ._tz import utcnow, utcfromtimestamp
from .utils import (
    check_pid,
    pathname2url,
    run_sync,
    unix_socket_in_use,
    url_escape,
    url_path_join,
    urldecode_unix_socket_path,
    urlencode_unix_socket,
    urlencode_unix_socket_path,
    urljoin,
)
from .traittypes import TypeFromClasses

# Check if we can use async kernel management
try:
    from jupyter_client import AsyncMultiKernelManager
    async_kernel_mgmt_available = True
except ImportError:
    async_kernel_mgmt_available = False

# Tolerate missing terminado package.
try:
    from .terminal import TerminalManager
    terminado_available = True
except ImportError:
    terminado_available = False

from .simple_handlers import (
    DefaultHandler, ErrorHandler, ParameterHandler, RedirectHandler, TemplateHandler, TypescriptHandler
)


#-----------------------------------------------------------------------------
# Module globals
#-----------------------------------------------------------------------------

_examples = """
jupyter notebook                       # start the notebook
jupyter notebook --certfile=mycert.pem # use SSL/TLS certificate
jupyter notebook password              # enter a password to protect the server
"""

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

def random_ports(port, n):
    """Generate a list of n random ports near the given port.

    The first 5 ports will be sequential, and the remaining n-5 will be
    randomly selected in the range [port-2*n, port+2*n].
    """
    for i in range(min(5, n)):
        yield port + i
    for i in range(n-5):
        yield max(1, port + random.randint(-2*n, 2*n))

def load_handlers(name):
    """Load the (URL pattern, handler) tuples for each component."""
    mod = __import__(name, fromlist=['default_handlers'])
    return mod.default_handlers

#-----------------------------------------------------------------------------
# The Tornado web application
#-----------------------------------------------------------------------------



class NotebookPasswordApp(JupyterApp):
    """Set a password for the notebook server.

    Setting a password secures the notebook server
    and removes the need for token-based authentication.
    """

    description = __doc__

    def _config_file_default(self):
        return os.path.join(self.config_dir, 'jupyter_notebook_config.json')

    def start(self):
        from .auth.security import set_password
        set_password(config_file=self.config_file)
        self.log.info("Wrote hashed password to %s" % self.config_file)


def shutdown_server(server_info, timeout=5, log=None):
    """Shutdown a notebook server in a separate process.

    *server_info* should be a dictionary as produced by list_running_servers().

    Will first try to request shutdown using /api/shutdown .
    On Unix, if the server is still running after *timeout* seconds, it will
    send SIGTERM. After another timeout, it escalates to SIGKILL.

    Returns True if the server was stopped by any means, False if stopping it
    failed (on Windows).
    """
    from tornado import gen
    from tornado.httpclient import AsyncHTTPClient, HTTPClient, HTTPRequest
    from tornado.netutil import Resolver
    url = server_info['url']
    pid = server_info['pid']
    resolver = None

    # UNIX Socket handling.
    if url.startswith('http+unix://'):
        # This library doesn't understand our URI form, but it's just HTTP.
        url = url.replace('http+unix://', 'http://')

        class UnixSocketResolver(Resolver):
            def initialize(self, resolver):
                self.resolver = resolver

            def close(self):
                self.resolver.close()

            @gen.coroutine
            def resolve(self, host, port, *args, **kwargs):
                raise gen.Return([
                    (socket.AF_UNIX, urldecode_unix_socket_path(host))
                ])

        resolver = UnixSocketResolver(resolver=Resolver())

    req = HTTPRequest(url + 'api/shutdown', method='POST', body=b'', headers={
        'Authorization': 'token ' + server_info['token']
    })
    if log: log.debug("POST request to %sapi/shutdown", url)
    AsyncHTTPClient.configure(None, resolver=resolver)
    HTTPClient(AsyncHTTPClient).fetch(req)

    # Poll to see if it shut down.
    for _ in range(timeout*10):
        if not check_pid(pid):
            if log: log.debug("Server PID %s is gone", pid)
            return True
        time.sleep(0.1)

    if sys.platform.startswith('win'):
        return False

    if log: log.debug("SIGTERM to PID %s", pid)
    os.kill(pid, signal.SIGTERM)

    # Poll to see if it shut down.
    for _ in range(timeout * 10):
        if not check_pid(pid):
            if log: log.debug("Server PID %s is gone", pid)
            return True
        time.sleep(0.1)

    if log: log.debug("SIGKILL to PID %s", pid)
    os.kill(pid, signal.SIGKILL)
    return True  # SIGKILL cannot be caught


class NbserverStopApp(JupyterApp):
    version = __version__
    description="Stop currently running notebook server."

    port = Integer(DEFAULT_NOTEBOOK_PORT, config=True,
        help="Port of the server to be killed. Default %s" % DEFAULT_NOTEBOOK_PORT)

    sock = Unicode(u'', config=True,
        help="UNIX socket of the server to be killed.")

    def parse_command_line(self, argv=None):
        super().parse_command_line(argv)
        if self.extra_args:
            try:
                self.port = int(self.extra_args[0])
            except ValueError:
                # self.extra_args[0] was not an int, so it must be a string (unix socket).
                self.sock = self.extra_args[0]

    def shutdown_server(self, server):
        return shutdown_server(server, log=self.log)

    def _shutdown_or_exit(self, target_endpoint, server):
        print("Shutting down server on %s..." % target_endpoint)
        server_stopped = self.shutdown_server(server)
        if not server_stopped and sys.platform.startswith('win'):
            # the pid check on Windows appears to be unreliable, so fetch another
            # list of servers and ensure our server is not in the list before
            # sending the wrong impression.
            servers = list(list_running_servers(self.runtime_dir))
            if server not in servers:
                server_stopped = True
        if not server_stopped:
            sys.exit("Could not stop server on %s" % target_endpoint)

    @staticmethod
    def _maybe_remove_unix_socket(socket_path):
        try:
            os.unlink(socket_path)
        except (OSError, IOError):
            pass

    def start(self):
        servers = list(list_running_servers(self.runtime_dir))
        if not servers:
            self.exit("There are no running servers (per %s)" % self.runtime_dir)

        for server in servers:
            if self.sock:
                sock = server.get('sock', None)
                if sock and sock == self.sock:
                    self._shutdown_or_exit(sock, server)
                    # Attempt to remove the UNIX socket after stopping.
                    self._maybe_remove_unix_socket(sock)
                    return
            elif self.port:
                port = server.get('port', None)
                if port == self.port:
                    self._shutdown_or_exit(port, server)
                    return
        else:
            current_endpoint = self.sock or self.port
            print(
                "There is currently no server running on {}".format(current_endpoint),
                file=sys.stderr
            )
            print("Ports/sockets currently in use:", file=sys.stderr)
            for server in servers:
                print("  - {}".format(server.get('sock') or server['port']), file=sys.stderr)
            self.exit(1)


class NbserverListApp(JupyterApp):
    version = __version__
    description=_("List currently running notebook servers.")

    flags = dict(
        jsonlist=({'NbserverListApp': {'jsonlist': True}},
              _("Produce machine-readable JSON list output.")),
        json=({'NbserverListApp': {'json': True}},
              _("Produce machine-readable JSON object on each line of output.")),
    )

    jsonlist = Bool(False, config=True,
          help=_("If True, the output will be a JSON list of objects, one per "
                 "active notebook server, each with the details from the "
                 "relevant server info file."))
    json = Bool(False, config=True,
          help=_("If True, each line of output will be a JSON object with the "
                  "details from the server info file. For a JSON list output, "
                  "see the NbserverListApp.jsonlist configuration value"))

    def start(self):
        serverinfo_list = list(list_running_servers(self.runtime_dir))
        if self.jsonlist:
            print(json.dumps(serverinfo_list, indent=2))
        elif self.json:
            for serverinfo in serverinfo_list:
                print(json.dumps(serverinfo))
        else:
            print("Currently running servers:")
            for serverinfo in serverinfo_list:
                url = serverinfo['url']
                if serverinfo.get('token'):
                    url = url + '?token=%s' % serverinfo['token']
                print(url, "::", serverinfo['notebook_dir'])

#-----------------------------------------------------------------------------
# Aliases and Flags
#-----------------------------------------------------------------------------

flags = dict(base_flags)
flags['no-browser']=(
    {'NotebookApp' : {'open_browser' : False}},
    _("Don't open the notebook in a browser after startup.")
)
flags['pylab']=(
    {'NotebookApp' : {'pylab' : 'warn'}},
    _("DISABLED: use %pylab or %matplotlib in the notebook to enable matplotlib.")
)
flags['no-mathjax']=(
    {'NotebookApp' : {'enable_mathjax' : False}},
    """Disable MathJax

    MathJax is the javascript library Jupyter uses to render math/LaTeX. It is
    very large, so you may want to disable it if you have a slow internet
    connection, or for offline use of the notebook.

    When disabled, equations etc. will appear as their untransformed TeX source.
    """
)

flags['allow-root']=(
    {'NotebookApp' : {'allow_root' : True}},
    _("Allow the notebook to be run from root user.")
)

flags['autoreload'] = (
    {'NotebookApp': {'autoreload': True}},
    """Autoreload the webapp

    Enable reloading of the tornado webapp and all imported Python packages
    when any changes are made to any Python src files in Notebook or
    extensions.
    """
)

# Add notebook manager flags
flags.update(boolean_flag('script', 'FileContentsManager.save_script',
               'DEPRECATED, IGNORED',
               'DEPRECATED, IGNORED'))

aliases = dict(base_aliases)

aliases.update({
    'ip': 'NotebookApp.ip',
    'port': 'NotebookApp.port',
    'port-retries': 'NotebookApp.port_retries',
    'sock': 'NotebookApp.sock',
    'sock-mode': 'NotebookApp.sock_mode',
    'transport': 'KernelManager.transport',
    'keyfile': 'NotebookApp.keyfile',
    'certfile': 'NotebookApp.certfile',
    'client-ca': 'NotebookApp.client_ca',
    'notebook-dir': 'NotebookApp.notebook_dir',
    'browser': 'NotebookApp.browser',
    'pylab': 'NotebookApp.pylab',
    'gateway-url': 'GatewayClient.url',
})

#-----------------------------------------------------------------------------
# NotebookApp
#-----------------------------------------------------------------------------

class NotebookApp(ExtensionAppJinjaMixin, ExtensionApp):

    name = 'notebook'

    extension_url = "/notebook/tree"

    # Should your extension expose other server extensions when launched directly?
    load_other_extensions = True

    # Local path to static files directory.
    static_paths = [DEFAULT_STATIC_FILES_PATH]

    # Local path to templates directory.
    template_paths = DEFAULT_TEMPLATE_PATH_LIST

    version = __version__
    description = _("""The Jupyter HTML Notebook.

    This launches a Tornado based HTML Notebook Server that serves up an HTML5/Javascript Notebook client.""")

    jinja_environment_options = Dict(config=True, 
        help=_("Supply extra arguments that will be passed to Jinja environment.")
    )

    jinja_template_vars = Dict(
        config=True,
        help=_("Extra variables to supply to jinja templates when rendering."),
    )

    # -------------------------------------------------------------------------
    def initialize_templates(self):
        _template_path = self.template_paths
        if isinstance(_template_path, py3compat.string_types):
            _template_path = (_template_path,)
        template_path = [os.path.expanduser(path) for path in _template_path]

        jenv_opt = {"autoescape": True}
        jenv_opt.update(self.jinja_environment_options if self.jinja_environment_options else {})

        env = Environment(loader=FileSystemLoader(template_path), extensions=['jinja2.ext.i18n'], **jenv_opt)

        # If the user is running the notebook in a git directory, make the assumption
        # that this is a dev install and suggest to the developer `npm run build:watch`.
        base_dir = os.path.realpath(os.path.join(__file__, '..', '..'))
        dev_mode = os.path.exists(os.path.join(base_dir, '.git'))

        nbui = gettext.translation('nbui', localedir=os.path.join(base_dir, 'notebook/i18n'), fallback=True)
        env.install_gettext_translations(nbui, newstyle=True)
        env.install_null_translations(newstyle=False)
        env.install_gettext_callables(gettext.gettext, gettext.ngettext)

        if dev_mode:
            DEV_NOTE_NPM = """It looks like you're running the notebook from source.
    If you're working on the Javascript of the notebook, try running
    %s
    in another terminal window to have the system incrementally
    watch and build the notebook's JavaScript for you, as you make changes.""" % 'npm run build:watch'
            self.log.info(DEV_NOTE_NPM)

        template_settings = dict(
            notebook_template_paths=template_path,
            jinja_template_vars=self.jinja_template_vars,
            notebook_jinja_template_vars=self.jinja_template_vars,
            jinja2_env=env,
            notebook_jinja2_env=env,
        )
        self.settings.update(**template_settings)


    # -------------------------------------------------------------------------
    def initialize_settings(self):
        self.initialize_templates()


    # -------------------------------------------------------------------------
    def initialize_handlers(self):
        self.handlers.extend(
            [
                (r"/{}/default".format(self.name), DefaultHandler),
                (r"/{}/params/(.+)$".format(self.name), ParameterHandler),
                (r"/{}/template1/(.*)$".format(self.name), TemplateHandler),
                (r"/{}/redirect".format(self.name), RedirectHandler),
                (r"/{}/typescript/?".format(self.name), TypescriptHandler),
                (r"/{}/(.*)", ErrorHandler),
            ]
        )
        """Load the (URL pattern, handler) tuples for each component."""
        # Order matters. The first handler to match the URL will handle the request.
        handlers = []
        # load extra services specified by users before default handlers
        for service in self.settings['extra_services']:
            handlers.extend(load_handlers(service))
        handlers.extend(load_handlers('notebook.tree.handlers'))
        handlers.extend(load_handlers('notebook.notebook.handlers'))
        print(handlers)

        # Add new handlers to Jupyter server handlers.
        self.handlers.extend(handlers)


def list_running_servers(runtime_dir=None):
    """Iterate over the server info files of running notebook servers.

    Given a runtime directory, find nbserver-* files in the security directory,
    and yield dicts of their information, each one pertaining to
    a currently running notebook server instance.
    """
    if runtime_dir is None:
        runtime_dir = jupyter_runtime_dir()

    # The runtime dir might not exist
    if not os.path.isdir(runtime_dir):
        return

    for file_name in os.listdir(runtime_dir):
        if re.match('nbserver-(.+).json', file_name):
            with io.open(os.path.join(runtime_dir, file_name), encoding='utf-8') as f:
                info = json.load(f)

            # Simple check whether that process is really still running
            # Also remove leftover files from IPython 2.x without a pid field
            if ('pid' in info) and check_pid(info['pid']):
                yield info
            else:
                # If the process has died, try to delete its info file
                try:
                    os.unlink(os.path.join(runtime_dir, file_name))
                except OSError:
                    pass  # TODO: This should warn or log or something
#-----------------------------------------------------------------------------
# Main entry point
#-----------------------------------------------------------------------------

main = launch_new_instance = NotebookApp.launch_instance
