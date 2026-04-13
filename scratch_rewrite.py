import re
import emoji

def clean_markdown():
    with open('/Users/adzhumurat/PycharmProjects/ai_product_engineer/slides/lecture_07_cracking_dnn.md', 'r') as f:
        text = f.read()

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove bold/italic markup
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\_\_(.*?)\_\_', r'\1', text)
    
    # Also clean single asterisks if they are formatting and not list marks
    # Instead of complex regex, let's just make sure we don't break list entries (* ...)
    # A simple bold is often ** text **, we handled that.
    # Single * italic: *text*, etc.
    text = re.sub(r'(?<!\w)\*(?!\s)([^*]+)(?<!\s)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_(?!\s)([^_]+)(?<!\s)_(?!\w)', r'\1', text)

    # Fix some headers that had emojis embedded
    text = re.sub(r'#+\s*(\d)\.?\s*', r'### \1. ', text)
    text = re.sub(r'#+\s*📌\s*', r'### ', text)
    text = re.sub(r'🔹\s*', r'', text)
    text = re.sub(r'💡\s*', r'', text)
    text = re.sub(r'✅\s*', r'- ', text)
    text = re.sub(r'🔥\s*', r'', text)
    text = re.sub(r'🚀\s*', r'', text)

    # Convert things like `### 1. Conv2D` or `## 1.` to a consistent `##` or `###`
    # Let's say all `### ` under `## ` without other things are fine, but let's change `### ` to `## ` if it's the main items.
    text = re.sub(r'^###\s+', '## ', text, flags=re.MULTILINE)
    
    with open('/Users/adzhumurat/PycharmProjects/ai_product_engineer/slides/lecture_07_cracking_dnn.md', 'w') as f:
        f.write(text)

if __name__ == "__main__":
    clean_markdown()
