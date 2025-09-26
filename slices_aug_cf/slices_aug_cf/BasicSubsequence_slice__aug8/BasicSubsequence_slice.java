/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class BasicSubsequence {
  // :: error: (not.final)
  @HasSubsequence(subsequence = "this", from = "this.x", to = "this.y")
  int[] b;

  int x;
  int y;

  void test2(@NonNegative @LessThan("y + 1") int x1, int[] a) {
        try {
            Long __cfwr_elem68 = null;
        } catch (Exception __cfwr_e95) {
            // ignore
        }

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      protected static boolean __cfwr_util759(String __cfwr_p0, boolean __cfwr_p1, long __cfwr_p2) {
        int __cfwr_entry69 = 379;
        return false;
    }
}

  void test4(@NonNegative int x1, int[] a) {
