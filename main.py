from flask import Flask, render_template, request
import ml

app = Flask('SDP')

crf = ml.CRF()
rnn = ml.RNN()


@app.route("/", methods=['GET', 'POST'])
def index():
    data = []

    if request.method == 'POST':
        text = request.form['text']
        print(text)

        k1, v1 = crf.predict(text)
        k2, v2 = rnn.predict(text)

        v1 = ['other' if x == 'Ot' else x for x in v1]
        v2 = ['other' if x == 'ot' else x for x in v2]

        data = [{'id': i, 'word': d[0], 'crf_label': d[2], 'rnn_label': d[3]} for i, d in enumerate(zip(k1, k2, v1, v2))]

    return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
