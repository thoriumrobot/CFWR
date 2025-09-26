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
        return null;

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      public static boolean __cfwr_handle348(Long __cfwr_p0, int __cfwr_p1, Object __cfwr_p2) {
        for (int __cfwr_i65 = 0; __cfwr_i65 < 9; __cfwr_i65++) {
            if (true && false) {
            return null;
        }
        }
        Integer __cfwr_result12 = null;
        Integer __cfwr_node73 = null;
        for (int __cfwr_i94 = 0; __cfwr_i94 < 2; __cfwr_i94++) {
            boolean __cfwr_result27 = ((248 - 540L) - (80.26f ^ -92.12f));
        }
        return true;
    }
}

  void test4(@NonNegative int x1, int[] a) {
