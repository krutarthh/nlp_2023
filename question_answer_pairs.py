


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import json
import os
from langdetect import detect


articles_dir = "D:\COVID-19-research\document_parses\pdf_json\\"
question_answer_data = {}
parser = argparse.ArgumentParser()
parser.add_argument('--num', '-n', help="number of documents to process", type=int, default=0)
parser.add_argument('--start', '-s', help="the starting index of document to parse", type=int, default= 0)

tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")

def main():
    args = parser.parse_args()
    n = args.num
    start = args.start
    try:
        with open("qa_pairs.json", "r") as outfile:
            question_answer_data = json.load(outfile)
    except:
        question_answer_data={}

    print("Current number of questions: ", len(question_answer_data.keys()))

    process_documents(n, start, question_answer_data)

    print("Final number of questions: ", len(question_answer_data.keys()))

    with open("qa_pairs.json", "w") as outfile:
        json.dump(question_answer_data, outfile)


def process_documents(num_docs, start, question_answer_data):
    # print(os.listdir(articles_dir))
    filenames = os.listdir(articles_dir)

    # Filter out only JSON files
    json_files = [file for file in filenames if file.endswith('.json')]
    print("Total number of json files: ", len(json_files))

    unloadable_invalid_files = []
    document_count = 0
    end = min(start + num_docs, len(json_files))
    print(f"Start:{start}, End:{end}" )
    for i in range(start, end):
        
        filename = json_files[i]
        filename_path = articles_dir +  filename
        try:
            with open(filename_path, 'r') as f:
                file = json.load(f)

                if "title" in file["metadata"]:
                    title = file["metadata"]["title"]
                else:
                    title = ""
                
                lang = detect(title.lower())
                if lang != "en":
                    print(f'File not in English: {title}')
                    continue
                    
                print("TITLE: ", title)
                context = ""
                for body_text in file["body_text"]: 
                    if "text" in body_text: 
                        context += body_text["text"].replace("\n", " ") 
                        # document["text"] += context + " "
                        if len(context.split()) >= 500:
                            question, answer = generate_qa_pairs(context, question_answer_data)
                            context = ""
                            print("Question: ", question)
                            print("Answer: ", answer)

                document_count += 1

                if document_count % 50 == 0:
                    print(f"Processed {document_count} documents...") 

        except Exception as e:
            print("ERROR: ", e)
            # If JSON decoding error occurs, add the file to the list
            unloadable_invalid_files.append(filename)
            # print(f"Error loading/reading {filename} due to error: {e}")
            continue
    # Print the list of files that couldn't be loaded
    if len(unloadable_invalid_files) > 0:
        print("Files that couldn't be loaded:")
        for file_name in unloadable_invalid_files:
            print(file_name)
            # subprocess.run(['rm', file_name])
        

    else:
        print("All JSON files were successfully loaded.")

    print("Total number of documents processed", document_count)



def generate_qa_pairs(context, question_answer_data):
    inputs = tokenizer(context, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    question, answer = question_answer.split(tokenizer.sep_token)

    question = question.strip()
    answer = answer.strip()
    if question not in question_answer_data:
        question_answer_data[question] = answer

    return question, answer

if __name__ == "__main__":
    main()

