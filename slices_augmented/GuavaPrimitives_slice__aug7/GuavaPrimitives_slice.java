/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class GuavaPrimitives_slice {
    @Positive
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        Long __cfwr_temp63 = null;

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

    static boolean __cfwr_aux342(Character __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i5 = 0; __cfwr_i5 < 6; __cfwr_i5++) {
            return null;
        }
        if ((255L | ('2' % -60.74)) && true) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 9; __cfwr_i54++) {
            return ((808L & 12.77) & null);
        }
        }
        return false;
    }
}