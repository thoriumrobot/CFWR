// This is a simplified version of a bug from JFreeChart that was confirmed by developers.

public class AddMonths {

    /*@ public invariant LAST_DAY_OF_MONTH.length == 13; */

    static final int [] LAST_DAY_OF_MONTH =
    {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

    /*@ public normal_behavior
      @ requires baseYear >= 1900 && baseYear <= 9999 && baseMonth >= 1 && baseMonth <= 12;
      @ ensures true;
      @*/
    public static void addMonths(int months, int baseYear, int baseMonth) {
        int mm = (12 * baseYear + baseMonth + months - 1) % 12 + 1;
    	int lastDayOfMonth = lastDayOfMonth(mm);
    }

    /*@ public normal_behavior
      @ requires month >= 1 && month <= 12;
      @ ensures true;
      @*/
    public static int lastDayOfMonth(int month) {
        final int result = LAST_DAY_OF_MONTH[month];
	    return result;
    }
}
