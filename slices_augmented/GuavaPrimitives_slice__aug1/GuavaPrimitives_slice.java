/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class GuavaPrimitives_slice {
    @Positive
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        if (false && false) {
            return null;
        }

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

    public static String __cfwr_func130(float __cfwr_p0, Double __cfwr_p1) {
        return null;
        return "value14";
    }
    Long __cfwr_proc418() {
        float __cfwr_entry84 = (true - null);
        try {
            try {
            return null;
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        for (int __cfwr_i37 = 0; __cfwr_i37 < 9; __cfwr_i37++) {
            return -725;
        }
        return null;
    }
}