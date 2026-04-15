"""Entry point: python -m dashboard"""

from dashboard.app import create_app

app = create_app()
app.run(debug=True, port=5050)
