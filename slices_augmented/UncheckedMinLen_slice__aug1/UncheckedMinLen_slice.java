/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UncheckedMinLen_slice {
    @Positive
  void addToNonNegative(@NonNegative int l, Object v) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }

    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToPositive(@Positive int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    @Positive
  void addToUnboundedIntRange(@IntRange(from = 0) int l, Object v) {
    // :: error: (assignment)
    @Positive
    Object @MinLen(100) [] o = new Object[l + 1];
    @Positive
    o[99] = v;
    @Positive
  }

    public static Character __cfwr_proc739(boolean __cfwr_p0) {
        while (true) {
            Long __cfwr_elem97 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}