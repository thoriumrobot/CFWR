/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class PlusPlusBug_slice {
    @Positive
  void test(@LTLengthOf("array") int x) {
        return null;

    // :: error: (unary.increment)
    @Positive
    x++;
    // :: error: (unary.increment)
    @Positive
    ++x;
    // :: error: (assignment)
    @Positive
    x = x + 1;
    @Positive
  }

    protected Character __cfwr_process420(Object __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        short __cfwr_val50 = (null & (687L << 45.08));
        return null;
        return null;
    }
}