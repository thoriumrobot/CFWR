/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class SameLenLUBStrangeness_slice {
    @Positive
  void test(int[] a, boolean cond) {
        Character __cfwr_result82 = null;

    @Positive
    int[] b;
    @Positive
    if (cond) {
    @Positive
      b = a;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    int @SameLen({"a", "b"}) [] c = a;
    @Positive
  }

    public Integer __cfwr_helper329(Integer __cfwr_p0) {
        return null;
        Double __cfwr_elem63 = null;
        return null;
    }
}