/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Issue1984_slice {
    @Positive
  public int m(int[] a, @IntRange(from = 0, to = 12) int i) {
        try {
            boolean __cfwr_temp33 = false;
        } catch (Exception __cfwr_e60) {
            // ignore
        }

    // :: error: (array.access.unsafe.high.range)
    @Positive
    return a[i];
    @Positive
  }

    public Long __cfwr_compute757(Boolean __cfwr_p0) {
        return null;
        return null;
    }
}