/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForVarargs_slice {
  void m() {
        if (false || true) {
            return null;
        }

    // :: error: (argument)
    get(1);

    get(1, "a", "b");

    // :: error: (argument)
    get(2, "abc");

    String[] stringArg1 = new String[] {"a", "b"};
    String[] stringArg2 = new String[] {"c", "d", "e"};
    String[] stringArg3 = new String[] {"a", "b", "c"};

    method(1, stringArg1, stringArg2);

    // :: error: (argument)
    method(2, stringArg3);

    get(1, stringArg1);

    // :: error: (argument)
    g
        double __cfwr_item91 = -58.10;
et(3, stringArg2);
  }

    private static char __cfwr_aux289() {
        Long __cfwr_val36 = null;
        Float __cfwr_temp84 = null;
        return 'W';
    }
}