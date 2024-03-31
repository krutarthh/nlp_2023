// Function to fetch JSON data
function fetchData(url) {
    return fetch(url)
        .then(response => response.json())
        .catch(error => {
            // Handle any errors that occur during fetching
            console.error(error);
        });
}

// Function to fill questions in a select element
function fillQuestions() {
    var questionSelect = document.getElementById("questions");

    fetchData('covid_related_qa_pairs.json')
        .then(data => {
            let questions = data;

            // Fill the select element with questions
            for (var question in questions) {
                var option = document.createElement("option");
                option.text = question;
                questionSelect.add(option);
            }
        });
}

// Function to display answer with typing effect
function showAnswer() {
    var questionSelect = document.getElementById("questions");
    var answerTextarea = document.getElementById("answer");
    var selectedQuestion = questionSelect.value;

    fetchData('covid_related_qa_pairs.json')
        .then(data => {
            var answers = data;
            answerTextarea.value = ""; // Clear the textarea before typing effect starts

            var answerText = answers[selectedQuestion] || ""; // Get the answer text
            var typingSpeed = 28; // Adjust the typing speed (in milliseconds)
            var index = 0;

            function typeAnswer() {
                if (index < answerText.length) {
                    answerTextarea.value += answerText.charAt(index);
                    index++;
                    setTimeout(typeAnswer, typingSpeed);
                }
            }

            typeAnswer(); // Start the typing effect
        });
}

// Query function to send data to the Hugging Face API for model inference
async function query(data) {
    const response = await fetch(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2",
        {
            headers: { Authorization: "Bearer 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" },
            method: "POST",
            body: JSON.stringify(data),
        }
    );
    const result = await response.json();
    return result;
}

// Initialize the page by filling questions when the window loads
window.onload = function() {
    fillQuestions();
};

// Fetch and process data for model inference when the page loads
async function fillquestion_userinput(input_question) {
    if (input_question==" "){
        return "Please enter a question.";
    }
    try {
        const data = await fetchData('covid_related_qa_pairs.json');
        const result = await query({
            "inputs": {
                "source_sentence": input_question,
                "sentences": Object.keys(data)
            }
        });

        let max = 0;
        let max_index = 0;
        let question = "";
        for (var i = 0; i < result.length; i++) {
            if (result[i] > max) {
                max = result[i];
                max_index = i;
            }
        }
        question = Object.keys(data)[max_index];

        let answer = data[question];

        return answer;

    } catch (error) {
        console.error(error);
    }
}

function showAnswer_userinput() {
    var questionInput = document.getElementById("user_question").value;
    var answerTextarea = document.getElementById("answer_user_question");
    answerTextarea.value = "";

    fillquestion_userinput(questionInput)
        .then(value => {
            let temp = value;
            var typingSpeed = 28;
            var index = 0;
            function typeAnswer() {
                if (index < temp.length) {
                    answerTextarea.value += temp.charAt(index);
                    index++;
                    setTimeout(typeAnswer, typingSpeed);
                }
            }

            typeAnswer(); // Start the typing effect
        });
}
