"""character.json の "describe"（下絵の指示に入る、このキャラの説明）の下書きを、Codex CLI が絵を見て書く。空の項目だけを
埋める。あとから人（か Claude）が直せる。

使い方: describe.py <プロジェクト>

  describe = {"character": "落ち着いた表情の少年", "hair": "毛先のはねた銀髪のショート、左に赤いメッシュ",
              "eyes": "左右で色の違う目（向かって左が赤、右が薄い灰色）", "skin": "色白",
              "outfit": "裏地の赤い黒の軍服風コート、…"}
（説明は日本語。下絵の指示は英語の文の中にこの日本語を入れるが、Codex も ChatGPT もそのまま読める。）
Codex CLI がないとき（手作業モード）は何もしない。そのときの指示は「髪」「目」のような言葉だけになるが、少し確実さが
落ちるだけで作れる。
"""
import os, sys, json, re, subprocess, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import platform_env

KEYS = ("character", "hair", "eyes", "skin", "outfit")
PROMPT = """Look at the attached character illustration and describe the character for an image-editing prompt. Write
each field as one short phrase IN JAPANESE (the person who checks it reads Japanese). Say left and right as seen by the
viewer (向かって左 / 向かって右). Reply with ONLY a JSON object, no other text:
{"character": "who it is - e.g. 落ち着いた表情の少年",
 "hair": "the hair - length, style, colour, streaks - e.g. 毛先のはねた銀髪のショート、左に赤いメッシュ",
 "eyes": "the eyes - colour of each, anything special",
 "skin": "the skin tone - e.g. 色白",
 "outfit": "the clothing and accessories, from head to feet, separated by 、"}"""


def main():
    P = C.open_project()
    have = dict(P.cfg.get("describe", {}))
    if all(have.get(k) for k in KEYS):
        print("説明: もう全部書いてあるので、何もしません")
        return
    codex = platform_env.codex_bin()
    if not codex:
        print("説明: Codex CLI がないので下書きできません。空欄のままか、自分で書いてください")
        return
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    gen = P.p("gen")
    os.makedirs(os.path.join(gen, "in"), exist_ok=True)
    os.makedirs(os.path.join(gen, "logs"), exist_ok=True)
    im = Image.open(P.p("src", "art.png")).convert("RGBA")
    s = 1536.0 / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(os.path.join(gen, "in", "describe_in.png"))
    out_txt = os.path.join(gen, "logs", "describe.answer.txt")
    cmd = [codex, "exec", "-C", gen, "--skip-git-repo-check", "--sandbox", "read-only", "-c", 'model_reasoning_effort="low"',
           "-i", os.path.join("in", "describe_in.png"), "-o", out_txt, "-"]
    with open(os.path.join(gen, "logs", "describe.log"), "w", encoding="utf-8", errors="replace") as lg:
        subprocess.run(cmd, input=PROMPT, text=True, encoding="utf-8", errors="replace", cwd=gen, stdout=lg, stderr=subprocess.STDOUT,
                       timeout=600)
    text = open(out_txt, encoding="utf-8", errors="replace").read() if os.path.exists(out_txt) else ""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        print("説明: Codex から返事がありませんでした（gen/logs/describe.log を見てください）")
        sys.exit(1)
    try:
        ans = json.loads(m.group(0))
    except json.JSONDecodeError:
        print("説明: Codex の返事を読み取れませんでした:", text[:300])
        sys.exit(1)
    for k in KEYS:
        if not have.get(k) and isinstance(ans.get(k), str) and ans[k].strip():
            have[k] = ans[k].strip()
    P.cfg["describe"] = have
    P.save_cfg()
    names = {"character": "どんなキャラか", "hair": "髪", "eyes": "目", "skin": "肌", "outfit": "服・持ち物"}
    for k in KEYS:
        print(f"{names[k]}: {have.get(k, '')}")


if __name__ == "__main__":
    main()
