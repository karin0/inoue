t0 = perf();

Redirecting {
  redir:;

  // Args. Set them before invoking us.
  text ?= 'Lorem ipsum dolor sit amet.';
  hook ?= 'echo hook: [$(cat -)]';
  stat ?= 0;

  text = cleanup(text);
  info = { bold(len(text)); ' chars to hook '; code(hook) };
  info;

  // Start a scope (`@`) to avoid name collisions.
  time = @ { t ↦
    r = {
      t >= 1 ? '%.3fs' % t :
        t >= 1e-3 ? '%.3fms' % (t * 1e3) :
          t >= 1e-6 ? '%.3fµs' % (t * 1e6) :
            '%.3fns' % (t * 1e9)
    };
    bold(r)
  };

  // `⇒` creates "closures" that captures the current scope, which is useful for
  // async callbacks. However, `↦` would also work here, since we are in the initial
  // root scope.
  rc1 = { r ⇒
    t2 = perf();
    times = {
      time(t1 - t0); ' + '; time(r.elapsed); ' + '; time(t2 - t1 - r.elapsed)
    };
    'Redirected '; info; ' in '; times; ' = '; time(t2 - t0);

    "r.returncode" ? ', status '; bold(r.returncode) !;
    '\n';

    "r.stdout" ? {
      "r.stderr" ?: 'stdout:\n';
      pre(r.stdout)
    };

    "r.stderr" ? {
      "r.stdout" ?: 'stderr:\n';
      pre(r.stderr)
    }
  };

  cnt = { stat ? 0 : 2 };
  view = { msg ⇒
    cnt < 3 ? {
      cnt = cnt + 1;
      body;
      stat ? {
        t3 = perf();
        times = { times; ' + '; time(t3 - t2) };
        t2 = $t3;
        'Sent '; msg; ' x '; cnt; ' in '; times; ' = '; time(t3 - t0)
      }
    }
  };

  rc2 = { body ⇒
    msg = '';
    w = { msg ⇒
      r = view(msg);
      r ? edit_message(r).then(w)
    };
    *w;
  };

  communicate(hook, text).then(rc1, rc2)

  // The async task is not scheduled until our entire rendering finishes, so we
  // take the time at the last moment.
} ...

t1 = "perf";
