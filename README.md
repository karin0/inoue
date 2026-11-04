# inoue

A single-user Telegram bot built around one idea: a message is a display that a
program rewrites in place. A command replies once and then keeps editing that
reply, so streamed subprocess output, a paged log viewer and a turn-based game
are the same mechanism seen from different angles.

The programs are written in a language of this repository's own, documented in
`render_core/README.md`. Sending `/render <source>` runs it, the result becomes
the message text, inline buttons become its input, and per-message state
persists across button presses. `examples/redirect_hook.m` is a document that
pipes text through an external command and reports the timings of each stage.

The closest analogy is a creative workshop inside a chat window. Its works are
documents, published by posting them to the channel, and commands, each one
function whose parameter annotations tell `Route` what to pass in. Works build
on each other, and the analogy stops at authorship, since `USER_ID` is the only
author.

## Layout

Three packages, with imports flowing in one direction.

`render_core/` is the language. It knows nothing about Telegram and runs
standalone through `python -m render_core.cli`.

`bot/` is the Telegram layer, described in `bot/README.md`. It does not import
`inoue`: `bot/env.py` declares a `Driver` protocol that `inoue/driver.py`
implements and registers, which keeps this deployment's chat ids, database and
commands out of `bot/`.

`inoue/` is everything specific to this bot. Its render half spans five
modules. `render.py` handles commands, buttons and callbacks, `render_ctx.py`
holds one render's execution state and trust level, `render_context.py` holds
the variable mapping and the `pm.` store, `render_bridge.py` is the language's
standard library, and `render_lib.py` provides promises and background tasks.

## Commands as language functions

There is one registry. `@command` puts a `Route` into `bot.dispatch.commands`,
and `Bridge.exec` reaches the same dict through `reroute_cmd`. Before calling,
`Responder.capture` installs a buffer. Each `reply` inside the command still
reaches the chat, and its text is also appended to the buffer and handed back to
the caller as a value. Capture ignores `cached` as well, keeping the command
from editing the reply that the capturing context owns. A command written for
the chat is therefore callable from a document with no adaptation:

    /render exec('/voice ' + btoa(8434178615911931332)).then({ r ↦ edit_message(r) })

One registry is why `/yt`, `/voice`, `/rg` and the rest need no binding table
and cannot drift out of sync with the language.

## Documents

A document is a named program. Posting one to `CHAN_ID`, as `USER_ID` or as the
channel, saves it: `handle_render_doc` renders the post, and when the source
declares a name, `db.save_doc` stores it under the post's own message id.
Editing the post so that it declares no name deletes the document. The channel
is the editor and its history is the revision log, while the `Doc` table is the
index `/ls` reads, whose links point back at the original posts.

A trusted render can also load a document from a file. `DOC_OVERRIDE_DIR` names
a directory of `<name>.m` files that it prefers over the database, and
`DOC_SEARCH_PATH` names further directories that it reads when the database has
no such document. `/submit <name>` posts a file from the override directory to
the channel, which sends it through the ordinary save path, and moves the file
aside so that the saved document takes over.

A guest reaches a document only when its name starts with one of
`ALLOWED_GUEST_DOC_PREFIXES`.

## Variables

The language leaves `:=`, `+name` and `-name` to the mapping its host passes to
`Engine`. Here that mapping is `render_context.OverriddenDict`, which freezes a
key written by any of them, so the first value outlasts every later assignment
in the render. `create_data` puts context values such as `_user_id` and
`_chat_id` among the frozen keys before rendering.

Every name under `pm.` is stored in the database, so its value outlives the
render and every document shares it. Any document can change such a value, so
nothing trusted may depend on one. A frozen write to a `pm.` name stays local to
the render.

## Trust

Three levels, decided in three places.

`inoue/__main__.py` decides whether an update is handled at all. It passes when
the sender is `USER_ID`, when the chat is `CHAN_ID`, when the content reached
`GROUP_ID` as an automatic forward from `CHAN_ID`, or when the sender is listed
in `GUEST_USER_IDS`. Everything else is dropped with a warning, and an update
from a chat in `IGNORE_CHAT_IDS` is dropped silently before any of these
checks.

`RenderContext.__init__` decides whether a render is trusted, from `USER_ID` and
`CHAN_ID`. The builtins that `render_bridge.py` marks `@trusted`, such as
`system`, `read_file`, `write_file`, `communicate` and `evil`, are reachable only
from there, while `@public` ones are reachable by anyone whose update survived
the first level.

A document may call `escalate()` to run trusted after a guest expanded it, and
the call succeeds only in a render of a saved document. It is a claim by the
document's author that every trusted builtin the document reaches does the same
thing whoever expanded it. A guest who presses a button can send any callback
data, so every flag, `_btn`, `_mem` and `_state` is guest input, and so is every
`pm.` name. An escalating document passes none of them to a trusted builtin and
never uses them to decide whether to call one. A flag may not name one of the
`HOST_KEYS` in `render.py`, so the context values the host writes, such as
`_user_id` and `_trusted`, are genuine or absent. Nothing verifies the claim.

Inside the language the names `os` and `sys` are bound to the `Bridge` object,
so source reaching for `os.system` finds the guarded method. simpleeval refusing
attribute names that start with an underscore is what keeps the real modules out
of reach. `evil` evaluates Python with the real `os` and `sys` in scope, which is
one reason it is trusted.

## Claude in other chats

When `USER_ID` mentions the bot in a chat the bot is not a member of, Telegram
delivers a guest message, and `inoue/ai.py` answers the text with Claude Code
through claude-agent-sdk, on the subscription that the system `claude` is logged
in with. Commands keep their usual handling. So do a bare mention, a URL alone
or followed by `q`, the parameters of a voice conversion after a URL or in reply
to audio or video, and the `rembg` of a sticker conversion, except in a reply to
a text of the bot, where a message with text or media continues the
conversation and takes its media to Claude as well. The answer starts as a status
line that names the running tool and becomes the answer when the turn ends.

Each chat has one session, recorded in the `AiSession` table and run in
`AI_CWD`, and `/clear` sent in the chat starts a new one. The session runs in
Claude Code's restricted mode, which loads no settings or `CLAUDE.md` file by
itself, and inoue passes it `AI_CWD/.claude/settings.json` so that the hooks
there can record the session for `USER_ID` to audit. Its tools search and fetch
the web, read the whole chat over MTProto as `USER_ID` through
`inoue/mtproto.py`, and run commands through `inoue/shell.py`. The tools stay
the same in every turn, since a change in them would invalidate the prompt
cache, and the shell tool runs a command only in a turn whose text starts with
`!`. `USER_ID` should send `!` only after reading what that turn can reach,
because text that others wrote in the chat could otherwise make the session act
on the machine.

Each prompt is a JSON object that names the chat, the bot and the time beside the
message and the message it replies to, since a guest message carries no other
context, and it states whether the shell tool works in this turn. The chat tool
returns JSON too. Every name and text that people wrote stays inside a JSON
string, and a `role` field marks the bot and `USER_ID` by their sender ids, so no
text or display name can forge a message or a role.

Each command runs in a transient systemd user scope with a time limit, so systemd
kills every process it started, including those started with `setsid`, even when
inoue has crashed, and inoue stops the scope as soon as the command's shell
exits. Of inoue's environment, which holds the bot token, the command receives
only the variables that describe the user's paths, locale and session bus.

inoue must be the only process that resumes a session, since two processes
resuming one session fork its transcript without an error. A session idle for
most of the prompt cache's hour compacts a large context while the cache still
holds it and closes, so the next turn resumes from the summary.

The chat tools need a Telethon session file logged in as `USER_ID`, which this
command creates once, asking for the login code on the terminal:

    set -a; . ./.env; set +a; uv run python -c "import os; from telethon.sync import TelegramClient; TelegramClient(os.environ['TELETHON_SESSION'], int(os.environ['TELETHON_API_ID']), os.environ['TELETHON_API_HASH']).start()"

## Configuration

Every value is read from the environment, and a missing required one crashes
startup. Concrete values live in a gitignored `.env`.

| Required | |
| --- | --- |
| `ME` | the bot's name, whose lowercase form names the default database file and the pid file |
| `TELEGRAM_BOT_TOKEN` | bot token |
| `USER_ID` | the one user this bot serves |
| `CHAN_ID` | channel where documents are authored and saved |
| `GROUP_ID` | group that receives the channel's automatic forwards and the quiet notifications |
| `MEDIA_STAGING_CHAT_ID` | chat where media is uploaded to obtain a file id |

| Optional | |
| --- | --- |
| `DB_FILE` | SQLite path, `<me>.db` in the working directory by default |
| `LOG_THREAD_ID`, `MEDIA_STAGING_MESSAGE_THREAD_ID` | forum topic ids for the notifications sent to `USER_ID` and for media staging |
| `GUEST_USER_IDS`, `IGNORE_CHAT_IDS` | comma-separated id lists of guests and of chats to ignore |
| `TRUSTED_IDS` | chats besides `USER_ID`, `CHAN_ID` and `GROUP_ID` whose command menu lists every command |
| `ALLOWED_GUEST_DOC_PREFIXES` | document name prefixes a guest may expand |
| `DOC_OVERRIDE_DIR`, `DOC_SEARCH_PATH` | file lookup for documents, the second colon-separated |
| `RG_CWD`, `UPDATE_CWD` | working directories for `/rg` and `/update` |
| `AI_CWD` | working directory of Claude's sessions and commands, which enables Claude and must hold `.claude/settings.json` |
| `AI_PROMPT_FILE` | a text file appended to Claude's system prompt |
| `TELETHON_SESSION`, `TELETHON_API_ID`, `TELETHON_API_HASH` | the Telethon session file and the API credentials from my.telegram.org, all three enabling Claude's chat tools |
| `YT_DLP_COOKIE_FILE` | cookies for yt-dlp |
| `MOTTO_FILE`, `SENTENCES_BUNDLE_DIR`, `HITOKOTO_TYPES`, `HITOKOTO_BANNED` | sources for `/greet` and `hitokoto()` |
| `LOG_FILTER_WORDS` | space-separated substrings that drop a log record at INFO or below |
| `LOCAL_SERVER`, `LOCAL_MODE` | a self-hosted Bot API server and its `--local` mode, which raises the upload limit from 50 MB to 2000 MB |
| `DEBUG`, `TRACE`, `TRACE_QUIET` | the first two take `1` and raise logging to that level, `TRACE` also traces the engine, and `TRACE_QUIET=1` trims that trace |

A module named `conf` on the import path may define `handle_help`, which then
replaces the default `/help`.

## External programs

`ffmpeg` and `ffprobe` for audio and video, `rg` for `/rg`, `lottie_to_gif.sh`
on `PATH` for animated sticker conversion, `bash` for `/run`, `sh` for the
`system` and `communicate` builtins, a `run.sh` in `UPDATE_CWD` for
`/update`, and `claude` and `systemd-run` with a running systemd user instance
for Claude in other chats.

## Running

Python 3.14 and uv. `uv run python -m inoue` starts the bot, which polls for
updates and holds `<me>.pid` in the working directory.

`uv sync --all-extras` prepares the tree, and `.github/workflows/ci.yml` runs
ruff check, ruff format, pyright and pytest over it.
`render_core/test_render.py` covers the language and `tests/` covers the bot.
The BPM detection tests call `ffmpeg`.
