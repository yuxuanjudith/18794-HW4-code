# question 4b, 4c
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-1.7B"

INSTRUCTION = ("Answer with only the name of the place where the person was born "
               "(a city or a country). Do not write a full sentence or any explanation.")


def load_data(path):
    questions, answers = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q, a = line.split('\t')
            questions.append(q)
            answers.append(a)
    return questions, answers


def rephrase(q):
    name = q[len('Where was '):-len(' born?')]
    return 'What is the birthplace of ' + name + '?'


def build_messages(q, use_prompt):
    if use_prompt:
        return [{"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": q}]
    return [{"role": "user", "content": q}]


def parse(output):
    text = output.strip().split('\n')[0]
    text = text.split(',')[0]
    return text.strip().strip('.').strip()


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--eval_corpus_path', default='birth_dev.tsv')
    argp.add_argument('--template', choices=['where', 'birthplace'], default='where')
    argp.add_argument('--use_prompt', action='store_true')
    argp.add_argument('--batch_size', type=int, default=16)
    argp.add_argument('--max_new_tokens', type=int, default=64)
    argp.add_argument('--outputs_path', default=None)
    args = argp.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()

    questions, answers = load_data(args.eval_corpus_path)
    if args.template == 'birthplace':
        questions = [rephrase(q) for q in questions]

    prompts = [tokenizer.apply_chat_template(build_messages(q, args.use_prompt),
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
               for q in questions]

    outputs = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[i:i + args.batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc,
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, enc['input_ids'].shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    if args.use_prompt:
        preds = [parse(o) for o in outputs]
    else:
        preds = [o.strip() for o in outputs]

    exact = sum(p == a for p, a in zip(preds, answers))
    contains = sum(a.lower() in o.lower() for o, a in zip(outputs, answers))
    total = len(answers)

    print(f'model: {MODEL_NAME} | template: {args.template} | use_prompt: {args.use_prompt}')
    print(f'Exact match: {exact} out of {total}: {exact / total * 100:.2f}%')
    print(f'Answer contained in output: {contains} out of {total}: {contains / total * 100:.2f}%')
    for q, o, a in list(zip(questions, outputs, answers))[:5]:
        print('-' * 60)
        print('Q:   ', q)
        print('GOLD:', a)
        print('OUT: ', o.strip().replace('\n', ' '))

    if args.outputs_path is not None:
        with open(args.outputs_path, 'w', encoding='utf-8') as f:
            for q, o, p, a in zip(questions, outputs, preds, answers):
                f.write(f"{q}\t{a}\t{p}\t{o.strip()!r}\n")


if __name__ == '__main__':
    main()