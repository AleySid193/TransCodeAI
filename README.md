# 💻 TransCodeAI: Pseudo Code ↔ C++ Converter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful AI-powered tool that seamlessly converts between Pseudo Code and C++ programming language using state-of-the-art transformer models.

## 🌟 Features

- **Bidirectional Conversion**: Convert in both directions:
  - Pseudo Code → C++
  - C++ → Pseudo Code
- **Smart Processing**: Utilizes advanced transformer models for accurate conversion
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Real-Time Conversion**: Instant results with proper formatting
- **Error Handling**: Robust error detection and informative messages

## 🚀 Demo

Access the live demo: [TransCodeAI Demo]([https://transcodai.streamlit.app](https://transcodeai.streamlit.app/))

## 📋 Requirements

- Python 3.8 or higher
- Streamlit
- PyTorch
- NumPy
- Python-Math

## 💾 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transcodai.git
cd transcodai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run converter.py
```

## 🎯 Usage

1. Select the conversion direction using the tabs:
   - "Pseudo Code → Code" for converting pseudo code to C++
   - "Code → Pseudo Code" for converting C++ to pseudo code

2. Enter your input code in the text area

3. Click the "Convert" button to see the results

### Example Conversions

#### Pseudo Code to C++
```
Input (Pseudo Code):
BEGIN
    READ number
    IF number > 0 THEN
        PRINT "Positive"
    ENDIF
END

Output (C++):
#include <iostream>
using namespace std;

int main() {
    int number;
    cin >> number;
    if (number > 0) {
        cout << "Positive";
    }
    return 0;
}
```

#### C++ to Pseudo Code
```
Input (C++):
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}

Output (Pseudo Code):
FUNCTION factorial(n)
    IF n <= 1 THEN
        RETURN 1
    ENDIF
    RETURN n * factorial(n-1)
END FUNCTION
```

## 🛠️ Technical Details

- **Model Architecture**: Transformer-based Sequence-to-Sequence model
- **Training Data**: Curated dataset of pseudo code and C++ pairs
- **Performance**: High accuracy with proper code structure preservation
- **Limitations**: Currently optimized for single-function/algorithm conversion

## 👥 Contributors

- Soban
- Ali

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⭐ Support

If you find this project helpful, please give it a star on GitHub!

---
Dev: Soban & Ali 
