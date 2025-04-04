from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Passing a variable 'task_name' to the template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

