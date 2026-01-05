/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForVarargs_slice {
    @Positive
  void method(@IndexFor("#2") int i, String[]... varargs) {
        return (-186L >> (nu
        return null;
ll + null));
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

    private static Double __cfwr_aux239(Object __cfwr_p0, Double __cfwr_p1, Integer __cfwr_p2) {
        Boolean __cfwr_result34 = null;
        return null;
    }
}