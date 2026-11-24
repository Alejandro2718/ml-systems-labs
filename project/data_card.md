# SMS Spam Collection Data Card

## Source and license

The SMS Spam Collection was collected by Tiago A. Almeida and José María Gómez Hidalgo and published in 2011. It combines public SMS messages from several sources, mostly from the UK and Singapore.

- [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84): Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository.
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

The pipeline downloads `sms+spam+collection.zip` from UCI. The archive SHA-256 is `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` and the `SMSSpamCollection` file SHA-256 is `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`. Both hashes are checked before the data is used.

## Size and format

Each line contains a label, a tab and the message text. The raw file has 5,574 rows: 4,827 ham and 747 spam (13.40%). All rows passed validation: the label is ham or spam, a tab separator exists and the text is not empty.

There are 403 exact duplicate rows. No message text appears with both labels, so no conflicting rows had to be removed ([label_conflicts.csv](results/label_conflicts.csv) is empty). After keeping the first copy of each text, 5,171 messages remain.

| Split | Messages | Spam | Spam share |
| --- | --- | --- | --- |
| Train | 3,619 | 457 | 12.63% |
| Validation | 776 | 98 | 12.63% |
| Test | 776 | 98 | 12.63% |

The split is stratified by label with seed 42. Deduplication happens before splitting, so the same text cannot appear in two splits; the pipeline checks this.

## Preprocessing

TF-IDF uses lowercase words and two-word phrases, sublinear term frequency, a minimum document frequency of 2 and at most 5,000 features. It is fitted on the training split only, then applied to validation and test.

## Limitations

The corpus is historical and mostly English, with regional abbreviations. It does not represent current SMS fraud, email or Spanish messages. Spam is the minority class, so accuracy alone is misleading; spam F1, precision and recall are reported. Removing duplicates lowers the apparent accuracy compared with studies that keep repeated spam templates in both training and test data.
