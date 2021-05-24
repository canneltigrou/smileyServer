from flask import Flask, render_template
#from flask import render_template


app = Flask(__name__)

@app.route('/')
def index():
  #return render_template('../test.html')
  return render_template('homepage-template.html', name = None) 


@app.route('/smiley')
def smiley():
  return render_template('smiley-template.html', name = None) #'<h2> Here will be a smiley </h2>'




# import io
# import random
# from flask import Response
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# @app.route('/plot.png')
# def plot_png():
#     fig = create_figure()
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig



# from smiley import displayDefaultSmiley;

# @app.route('/whatsnext/', methods=['POST'])
# def waiting():
#     # curVal=request.form['x']
#     nextVal = displayDefaultSmiley()
#     return render_template('post_response.html', nextVal=nextVal)


#@app.route('/my-link/')
#def my_link():
#  print ('I got clicked!')
#
#  return 'Click.'

if __name__ == '__main__': # execute only if in main page, not called.
  app.run(debug=True)