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

@app.route('/',methods=['GET','POST'])
def users():
    return jsonify (hello='This is a RSV database')

@app.route('/get')
def get(): 
    db = get_db()
    data = db.execute("select * from genebank").fetchall()
    return jsonify(data)

@app.route('/post/')
def post():
    db = get_db()
    acession = request.args.get("acession","RSVdb")
    country = request.args.get("country","RSVdb")
    year = request.args.get("year","RSVdb")
    g.db.execute("INSERT INTO experiment VALUES ('acession','country','year')")
       
    g.db.commit()
    data = db.execute("select * from genebank").fetchall()
    return jsonify(data)

