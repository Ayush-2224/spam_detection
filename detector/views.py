#from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .forms import EmailForm
from .models import SpamDetector

detector = SpamDetector()

def home(request):
    result = None
    if request.method == 'POST':
        form = EmailForm(request.POST)
        if form.is_valid():
            email_text = form.cleaned_data['email_text']
            prediction = detector.predict(email_text)
            result = 'Spam' if prediction == 1 else 'Not Spam'
    else:
        form = EmailForm()
    return render(request, 'detector/home.html', {'form': form, 'result': result})
