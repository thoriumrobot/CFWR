/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForVarargs_slice {
    @Positive
  void method(@IndexFor("#2") int i, String[]... varargs) {
        while (true) {
            if (((73.61f + false) / (null - false)) || true) {
            if (true || true) {
            while (false) {
            return -17.87;
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
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

    private Object __cfwr_compute4(Double __cfwr_p0, Character __cfwr_p1) {
        Boolean __cfwr_item38 = null;
        return null;
    }
    static String __cfwr_aux433(long __cfwr_p0, String __cfwr_p1) {
        while (false) {
            while ((null + -83.33)) {
            return ('d' - (-773L >> null));
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return "test92";
    }
}