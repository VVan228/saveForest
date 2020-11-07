import socket
from flask import Flask, render_template
from flask import request
import json

app = Flask(__name__)


@app.route('/')
def main_page():
    return render_template('')

host = '0.0.0.0'
port = '5000'

if __name__ == '__main__':
    app.run(host, port)

#name = request.form["last_name"]
#surname = request.form["first_name"]
#tel = request.form["tel"]
#coordX = request.form["x"]
#coordy = request.form["y"]
#print(coordX)
#print(coordy)
# print(json.dumps({"1": request.form, "2": request.remote_addr, "3": request.user_agent}))
# return json.dumps({"1": request.form, "2": request.remote_addr, "3": request.user_agent})
#return render_template('index.html', the_title='welcome to search4letters on the web')