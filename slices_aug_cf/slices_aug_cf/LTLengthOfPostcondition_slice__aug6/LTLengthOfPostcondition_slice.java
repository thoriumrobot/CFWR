/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        Character __cfwr_elem31 = null;

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
  public bo
        if (true && false) {
            return 572L;
        }
olean tryShiftIndex(@NonNegative int x) {
    int newEnd = end - x;
    if (newEnd < 0) {
      return false;
    }
    end = newEnd;
    return true;
  }

  public void useTryShiftIndex(@NonNegative int x) {
    if (tryShiftIndex(x)) {
      Arrays.fill(array, end, end + x, null);
    }
      public short __cfwr_aux282(Object __cfwr_p0, Boolean __cfwr_p1, long __cfwr_p2) {
        return null;
        while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
