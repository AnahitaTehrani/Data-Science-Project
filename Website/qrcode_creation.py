from qrcode import QRCode

# Daten, die im QR-Code codiert werden sollen, z.B. eine URL oder Text
data = "https://data-science-project-spotify.onrender.com"

# Erstellen des QR-Codes
qr = QRCode()
qr.add_data(data)
qr.make()
img = qr.make_image()

# Speichern als PNG-Datei
img.save("qr_code.png")