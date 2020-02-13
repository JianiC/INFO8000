from flask import Flask, escape, request, jsonify,g
import sqlite3 as sq1


DATABASE = 'RSV.db'
app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sq1.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def users():
    return jsonify (hello='This is a RSV database')

@app.route('/get')
def get(): 
    db = get_db()
    data = db.execute("select * from sequence").fetchall()
    return jsonify(data)

@app.route('/post', methods = ['POST'])
def post():
    db = get_db()
    seq = request.args.get("id","seq")
    g.db.execute("INSERT INTO sequence VALUES ((?,?)", [seq])
    g.db.commit()
    return redirect('/get')