/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RangeIndex_slice {
    @Positive
  void foo(@IntRange(from = 0, to = 11) int x, int @MinLen(10) [] a) {
        if (('6' & (-43.10 | 20.01f)) && false) {
            return -19.09f;
        }

    // :: error: (array.access.unsafe.high.range)
    @Positive
    int y = a[x];
    @Positive
  }

    private long __cfwr_util556() {
        return true;
        return 258L;
    }
}