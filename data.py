import pandas as pd

data = {
    "source": [
        "ನಮಸ್ಕಾರ",
        "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "ನನ್ನ ಹೆಸರು ರವಿ",
        "ನಿಮ್ಮ ಹೆಸರು ಏನು?",
        "ಇದು ನನ್ನ ಪುಸ್ತಕ",
        "ನನಗೆ ಕನ್ನಡ ಇಷ್ಟ",
        "ನಾವು ಶಾಲೆಗೆ ಹೋಗುತ್ತೇವೆ",
        "ಅವನು ನನ್ನ ಸ್ನೇಹಿತ",
        "ಇದು ಒಂದು ಚೆನ್ನಾದ ದಿನ",
        "ನಾನು ಭಾರತದಿಂದ ಬಂದಿದ್ದೇನೆ"
    ],
    "target": [
        "Hello",
        "How are you?",
        "My name is Ravi",
        "What is your name?",
        "This is my book",
        "I like Kannada",
        "We go to school",
        "He is my friend",
        "This is a beautiful day",
        "I am from India"
    ]
}

df = pd.DataFrame(data)

df.to_csv("data.csv", index=False, encoding="utf-8")

print("CSV file created successfully!")
