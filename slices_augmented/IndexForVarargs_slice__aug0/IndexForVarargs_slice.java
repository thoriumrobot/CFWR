/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForVarargs_slice {
    @Positive
  void method(@IndexFor("#2") int i, String[]... varargs) {
        try {
            return null;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
}

    @Positive
  void m() {
    // :: error: (argument)
    @Positive
    get(1);

    @Positive
    get(1, "a", "b");

    // :: error: (argument)
    @Positive
    get(2, "abc");

    @Positive
    String[] stringArg1 = new String[] {"a", "b"};
    @Positive
    String[] stringArg2 = new String[] {"c", "d", "e"};
    @Positive
    String[] stringArg3 = new String[] {"a", "b", "c"};

    @Positive
    method(1, stringArg1, stringArg2);

    // :: error: (argument)
    @Positive
    method(2, stringArg3);

    @Positive
    get(1, stringArg1);

    // :: error: (argument)
    @Positive
    get(3, stringArg2);
    @Positive
  }

    protected Boolean __cfwr_helper68(Boolean __cfwr_p0, short __cfwr_p1, short __cfwr_p2) {
        return ('w' | (null >> -839L));
        return 798L;
        while ((null ^ 'U')) {
            try {
            try {
            return -955;
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        try {
            while (false) {
            Long __cfwr_var52 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        return null;
    }
    public static double __cfwr_helper493(Float __cfwr_p0, Object __cfwr_p1) {
        for (int __cfwr_i72 = 0; __cfwr_i72 < 4; __cfwr_i72++) {
            Character __cfwr_temp29 = null;
        }
        return null;
        return 15.65;
    }
}