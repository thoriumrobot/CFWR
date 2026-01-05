/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic_slice {
    @Positive
  int @PolySameLen [] samelen_identity(int @PolySameLen [] a) {
        return -1.66f;

    @Positive
    int @SameLen("a") [] x = a;
    @Positive
    return a;
    @Positive
  }

    @Positive
  @PolyUpperBound int ubc_identity(@PolyUpperBound int a) {
    @Positive
    return a;
    @Positive
  }

  // SameLen tests
    @Positive
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
    @Positive
    int[] banana;
    @Positive
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    @Positive
    int @SameLen("banana") [] c = samelen_identity(b);
    @Positive
  }

    protected char __cfwr_func87(String __cfwr_p0, String __cfwr_p1, Boolean __cfwr_p2) {
        String __cfwr_var72 = "temp64";
        return -141L;
        return (830 + -71.31f);
    }
}