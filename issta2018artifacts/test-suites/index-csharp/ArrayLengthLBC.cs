using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class Date { }

public class ArrayLengthLBC {

    public static Date[] add_date(Date[] dates, Date new_date) {
        Date[] new_dates = new Date[dates.Length + 1];
        Array.Copy(dates, 0, new_dates, 0, dates.Length);
        new_dates[dates.Length] = new_date;
        Date[] new_dates_cast = new_dates;
        return (new_dates_cast);
    }
}
// a comment
