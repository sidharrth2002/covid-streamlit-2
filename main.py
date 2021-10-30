from multiapp import MultiPage
from pages import EDA, Clustering, Classification

app = MultiPage()
app.add_page("EDA", EDA.app)
app.add_page("Clusteing", Clustering.app)
app.add_page("Classification", Classification.app)

app.run()