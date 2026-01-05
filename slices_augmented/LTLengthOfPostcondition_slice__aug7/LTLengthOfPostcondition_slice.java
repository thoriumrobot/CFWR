/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLengthOfPostcondition_slice {
    @Positive
  public void shiftIndex(@NonNegative int x) {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 10; __cfwr_i43++) {
            if (false || (null * -182)) {
            try {
            return null;
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        }
        }

    @Positive
    int newEnd = end - x;
    @Positive
    if (newEnd < 0) throw new RuntimeException();
    @Positive
    end = newEnd;
    @Positive
  }

    @Positive
  public void useShiftIndex(@NonNegative int x) {
    // :: error: (argument)
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
    shiftIndex(x);
    @Positive
    Arrays.fill(array, end, end + x, null);
    @Positive
  }

    private Integer __cfwr_compute529() {
        Object __cfwr_temp98 = null;
        if (true && ((-424L + 705L) * -829L)) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 8; __cfwr_i32++) {
            return null;
        }
        }
        return null;
    }
}