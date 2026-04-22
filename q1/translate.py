# from transformers import pipeline
# from striprtf.striprtf import rtf_to_text
# import sacrebleu

# pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en")

# with open('./input.rtf', 'r') as f:
#     rtf = f.read()

# input_lines = [line.strip() for line in rtf_to_text(rtf).split('\n') if line.strip()]

# translations = []
# for line in input_lines:
#     result = pipe(line)
#     translations.append(result[0]['translation_text'])

# with open('./output.txt', 'w') as f:
#     f.write(output[0]['translation_text'])

# with open('./reference.rtf', 'r') as f:
#     ref_rtf = f.read()
# ref_lines = [line.strip() for line in rtf_to_text(ref_rtf).split('\n') if line.strip()]

# # bleu score now
# bleu = sacrebleu.corpus_bleu([output[0]['translation_text']], [[ref_text]])
# print(bleu.score)

from transformers import pipeline
from striprtf.striprtf import rtf_to_text
import sacrebleu

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-bn-en")

with open('./input.rtf', 'r', encoding='utf-8') as f:
    rtf = f.read()
input_lines = [line.strip() for line in rtf_to_text(rtf).split('\n') if line.strip()]

print(f"Translating {len(input_lines)} lines")
translations = []
for line in input_lines:
    result = pipe(line)
    translations.append(result[0]['translation_text'])

gen_text = "\n".join(translations)
with open('./output.txt', 'w', encoding='utf-8') as f:
    f.write(gen_text)

with open('./reference.rtf', 'r', encoding='utf-8') as f:
    ref_rtf = f.read()
ref_lines = [line.strip() for line in rtf_to_text(ref_rtf).split('\n') if line.strip()]

bleu = sacrebleu.corpus_bleu(translations, [ref_lines])
print(f"\nFinal BLEU Score: {bleu.score}")
