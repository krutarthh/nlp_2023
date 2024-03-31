import json
from transformers import pipeline

covid_related_qa_pairs = {}
discarded_qa_pairs = {}

with open("covid_related_qa_pairs.json", "r") as outfile:
    covid_related_qa_pairs = json.load(outfile)

with open("discarded_qa_pairs.json", "r") as outfile:
    discarded_qa_pairs = json.load(outfile)


print("CURRENT TOTAL NUM OF VALID QUESTIONS: ", len(covid_related_qa_pairs.keys()))
print("CURRENT TOTAL NUM OF DISCARDED QUESTIONS: ", len(discarded_qa_pairs.keys()))


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

candidate_labels = ['covid', 'pandemic', 'medical', 'other']

with open("qa_pairs.json", "r") as outfile:
    question_answer_data = json.load(outfile)

    for question in question_answer_data:
        answer = question_answer_data[question]

        # print(question, answer)

        if answer.find("<unk>") != -1 or question.find("<unk>") != -1:
            continue

        
        resp = classifier(question.lower(), candidate_labels)
        # print("RESP: ", resp)
        label = resp["labels"][0]
        score = resp["scores"][0]

        question = question.strip()
        answer = answer.strip()

        if question.lower().find('covid') != -1 or (label != "other" and score > 0.5):
            print("COVID-RELATED: ", question)
            if question not in covid_related_qa_pairs:
                covid_related_qa_pairs[question] = answer
        else:
            print("NOT COVID-RELATED: ", question)
            if question not in discarded_qa_pairs:
                discarded_qa_pairs[question] = answer
            
print("FINAL TOTAL NUM OF VALID QUESTIONS: ", len(covid_related_qa_pairs.keys()))
print("FINAL TOTAL NUM OF DISCARDED QUESTIONS: ", len(discarded_qa_pairs.keys()))

        
with open("covid_related_qa_pairs.json", "w") as outfile:
    json.dump(covid_related_qa_pairs, outfile)

with open("discarded_qa_pairs.json", "w") as outfile:
    json.dump(discarded_qa_pairs, outfile)