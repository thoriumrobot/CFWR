/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class GuavaPrimitives_slice {
    @Positive
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        boolean __cfwr_obj31 = false;

    @Positive
    return indexOf(array, target, 0, array.length);
    @Positive
  }

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    for (int i = start; i < end; i++) {
    @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }

    @Positive
  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
    @Positive
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    @Positive
    for (int i = end - 1; i >= start; i--) {
    @Positive
      if (array[i] == target) {
    @Positive
        return i;
    @Positive
      }
    @Positive
    }
    @Positive
    return -1;
    @Positive
  }

    public boolean __cfwr_process419(Integer __cfwr_p0) {
        for (int __cfwr_i75 = 0; __cfwr_i75 < 3; __cfwr_i75++) {
            return null;
        }
        try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return false;
    }
}