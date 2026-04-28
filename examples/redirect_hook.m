{
  redir:;

  // args
  text ?= 'Lorem ipsum dolor sit amet.';
  hook ?= 'echo hook: [$(cat -)]';

  text = "cleanup(text)";
  info = { "bold(len(text))"; ' chars to hook '; "code(hook)" };

  rc = { r ⇒
    body = {
      'Redirected '; info; ' in ';

      dt = {
        "r.elapsed >= 1" ? "'%.3fs' % r.elapsed" : "'%dms' % (r.elapsed * 1000)";
      };
      "bold(dt)";

      "r.returncode" ? ', status '; "bold(r.returncode)" !;
      '\n';

      "r.stdout" ? {
        "r.stderr" ?: 'stdout:\n';
        "pre(r.stdout)"
      };

      "r.stderr" ? {
        "r.stdout" ?: 'stderr:\n';
        "pre(r.stderr)"
      }
    };
    "edit_message(body)"
  };
  "communicate(hook, text).then(rc)";

  'Redirecting '; info; ' ...'
}
