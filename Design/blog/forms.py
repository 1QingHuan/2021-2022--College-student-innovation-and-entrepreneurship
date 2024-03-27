from django import forms
from blog.models import ArticleComment


class ArticleCommentForm(forms.Form):
    body = forms.CharField(required=True, label="",
                           error_messages={'required': '...',
                                           'min_length': '再输入点内容吧'},
                           min_length=2, widget=forms.Textarea(attrs={'placeholder': '路过匆匆，何不留下您的看法呢....'}))
    class Meta:
        model = ArticleComment
        fields = ['body,user_name']

    def clean_body(self):
        message = self.cleaned_data['body']
        if "fuck" in message:
            raise forms.ValidationError('请文明用语')
        return message
class historyinfo(forms.Form):
    kmnc=forms.CharField(max_length=20)
    nbc=forms.CharField(max_length=20)
    snac=forms.CharField(max_length=20)
    tknc=forms.CharField(max_length=20)


    def __str__(self):
        return self.name


