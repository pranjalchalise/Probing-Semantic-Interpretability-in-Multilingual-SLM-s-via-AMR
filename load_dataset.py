import penman
import pandas as pd

def load_amr_text(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        content = f.read().strip()
    entries = content.split("\n\n")

    for entry in entries:
        lines = entry.strip().split("\n")

        # extract the sentence
        snt_line = next((l for l in lines if l.startswith('# ::snt')), None)
        sentence = snt_line.replace('# ::snt', '').strip() if snt_line else None

        # extracting the language
        id_line = next((l for l in lines if l.startswith('# ::id')), None)
        lang = id_line.split('-')[-1] if id_line else None

        # AMR graph text
        amr_str = "\n".join(l for l in lines if not l.startswith('#')).strip()

        try:
            graph = penman.decode(amr_str)
        except:
            graph = None

        data.append({"sentence": sentence, "graph": graph, "lang": lang})

    return data


def extract_features(graph):
    if graph is None:
        return {"ARG0": 0, "ARG1": 0, "ARG2": 0, "neg": 0, "time": 0}

    roles = [role.lstrip(":") for (_, role, _) in graph.edges()]

    return {
        "ARG0": int("ARG0" in roles),
        "ARG1": int("ARG1" in roles),
        "ARG2": int("ARG2" in roles),
        "neg": int("polarity" in roles),
        "time": int(any(r in roles for r in ["time", "day", "month", "year"]))
    }


# dataset path
amr_path = './data/amrs-massive-val.txt'
dataset = load_amr_text(amr_path)

# convert to a pandas df
rows = []
for item in dataset:
    feats = extract_features(item["graph"])
    rows.append({
        "sentence": item["sentence"],
        "lang": item["lang"],
        **feats
    })

df = pd.DataFrame(rows)
print(df.sample(10).to_string())
