/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLengthOfPostcondition_slice {
    @Positive
  public void shiftIndex(@NonNegative int x) {
        double __cfwr_temp85 = ((true >> '5') | null);

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

    public static String __cfwr_aux940(Float __cfwr_p0) {
        for (int __cfwr_i64 = 0; __cfwr_i64 < 5; __cfwr_i64++) {
            if (((-5.20 & -780L) - '1') && true) {
            if (true || false) {
            return (true & 98.02f);
        }
        }
        }
        return "world54";
    }
}