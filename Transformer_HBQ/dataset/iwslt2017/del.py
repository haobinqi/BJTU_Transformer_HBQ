import re

def extract_text_from_xml(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 提取 <description> 或 <seg> 内容
    # segments = re.findall(r"<description>(.*?)</description>", text, flags=re.S)
    segments = re.findall(r"<seg[^>]*>(.*?)</seg>", text, flags=re.S)

    for seg in segments:
        seg = seg.strip()  # 去掉首尾空白和换行
        if seg:
            lines.append(seg)
    return lines
def val():
    # 读取英文和德文文件
    src_lines = extract_text_from_xml("test.en.xml")
    tgt_lines = extract_text_from_xml("test.de.xml")

    print(f"源句 {len(src_lines)} 条，目标句 {len(tgt_lines)} 条")

    # 检查是否一一对齐
    if len(src_lines) != len(tgt_lines):
        print("⚠️ 警告：源和目标数量不一致，请检查数据文件。")

    # 保存为平行语料，每条保持一行
    with open("test.en", "w", encoding="utf-8") as f:
        for line in src_lines:
            f.write(line + "\n")

    with open("test.de", "w", encoding="utf-8") as f:
        for line in tgt_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    val()

