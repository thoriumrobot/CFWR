/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class SLSubtyping_slice {
    @Positive
  void subtype(int @SameLen("#2") [] a, int[] b) {
        try {
            Long __cfwr_elem68 = null;
        } catch (Exception __cfwr_e95) {
            // ignore
        }

    @Positive
    int @SameLen({"a", "b"}) [] c = a;

    // :: error: (assignment)
    @Positive
    int @SameLen("c") [] q = {1, 2};
    @Positive
    int @SameLen("c") [] d = q;

    // :: error: (assignment)
    @Positive
    int @SameLen("f") [] e = a;
    @Positive
  }

    protected static boolean __cfwr_util759(String __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        int __cfwr_entry69 = 379;
        return false;
    }
}