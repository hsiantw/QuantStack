from pathlib import Path
import tempfile
import unittest

from database_browser import inspect
from market_data import connect


class DatabaseBrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'market.sqlite'
        with connect(self.path) as db, db:
            for symbol in ('AAA','BBB'):
                for day in range(1,6):
                    db.execute('INSERT INTO prices(symbol,date,close,source) VALUES (?,?,?,?)',(symbol,f'2026-10-{day:02d}',day,'Yahoo Finance'))
        db.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_overview_schema_and_filtered_pagination(self):
        meta=inspect(self.path,{})
        self.assertTrue(meta['read_only'])
        self.assertIn('prices',meta['tables'])
        self.assertIn('close',[c['name'] for c in meta['schemas']['prices']])
        q={'table':['prices'],'symbol':['aaa'],'start':['2026-10-02'],'end':['2026-10-04'],'limit':['2']}
        data=inspect(self.path,q)
        self.assertEqual(data['count'],3)
        self.assertEqual([r['date'] for r in data['rows']],['2026-10-04','2026-10-03'])
        self.assertTrue(data['has_more'])
        tail=inspect(self.path,dict(q,offset=['2']))
        self.assertEqual(tail['rows'][0]['date'],'2026-10-02')
        self.assertFalse(tail['has_more'])

    def test_filters_are_literals_and_invalid_identifiers_rejected(self):
        result=inspect(self.path,{'table':['prices'],'symbol':["AAA' OR 1=1--"]})
        self.assertEqual(result['count'],0)
        for q in [{'table':['prices; DROP TABLE prices']},{'table':['prices'],'status':['ok']},
                  {'table':['prices'],'limit':['1000000']},{'table':['prices'],'offset':['-1']},
                  {'table':['prices'],'start':['2026-12-01'],'end':['2026-01-01']},
                  {'table':['prices'],'end':['9999-12-31']}]:
            with self.subTest(q=q),self.assertRaises(ValueError):inspect(self.path,q)
        self.assertEqual(inspect(self.path,{'table':['prices']})['count'],10)

    def test_no_database_is_created_and_rows_remain_unchanged(self):
        missing=Path(self.temp.name)/'missing.sqlite'
        with self.assertRaises(ValueError):inspect(missing,{})
        self.assertFalse(missing.exists())
        before=self.path.read_bytes()
        inspect(self.path,{'table':['prices']})
        self.assertEqual(self.path.read_bytes(),before)


if __name__=='__main__': unittest.main()
