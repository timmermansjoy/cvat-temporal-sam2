from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("engine", "0106_add_interval_annotations"),
    ]

    operations = [
        migrations.AddField(
            model_name="trackedshape",
            name="source",
            field=models.CharField(
                choices=[
                    ("auto", "AUTO"),
                    ("semi-auto", "SEMI_AUTO"),
                    ("manual", "MANUAL"),
                    ("file", "FILE"),
                    ("consensus", "CONSENSUS"),
                ],
                default=None,
                max_length=16,
                null=True,
            ),
        ),
    ]
