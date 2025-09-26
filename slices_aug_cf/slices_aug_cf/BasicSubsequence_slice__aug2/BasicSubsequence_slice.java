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
        if (true && ('e' & 139)) {
            return null;
        }

    x = x1;
    // :: error: (to.not.ltel)
    b = a;
  }

  void test3(@NonNegative @LessThan("y") int x1, int[] a) {
    x = x1;
    // :: error: (to.not.ltel)
    b = a;
      public static Object __cfwr_func390(boolean __cfwr_p0, Integer __cfwr_p1, char __cfwr_p2) {
        return null;
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    public String __cfwr_helper927(Boolean __cfwr_p0, Boolean __cfwr_p1) {
        Long __cfwr_item92 = null;
        while (false) {
            if (true && ((-160 << false) - null)) {
            return null;
        }
            break; // Prevent infinite loops
        }
        byte __cfwr_node37 = null;
        return "hello13";
    }
}

  void test4(@NonNegative int x1, int[] a) {
