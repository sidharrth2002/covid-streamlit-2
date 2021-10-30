from multiapp import MultiPage
from pages import EDA, Clustering, Classification, Regression

app = MultiPage()
app.add_page("EDA", EDA.app)
app.add_page("Clustering", Clustering.app)
app.add_page("Regression", Regression.app)
app.add_page("Classification", Classification.app)

app.run()