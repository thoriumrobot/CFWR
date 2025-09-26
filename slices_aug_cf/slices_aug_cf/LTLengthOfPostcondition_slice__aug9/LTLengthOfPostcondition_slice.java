/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void useShiftIndex(@NonNegative int x) {
        Object __cfwr_obj75 = null;

    // :: error: (argument)
    Arrays.fill(array, end, end + x, null);
    shiftIndex(x);
    Arrays.fill(array, end, end + x, null);
  }

  @EnsuresLTLengthOfIf(expression = "end", result = true, targetValue = "array", offset = "#1 - 1")
  public boolean tryShiftIndex(@NonNegative int x) {
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
      public static short __cfwr_handle1(Character __cfwr_p0, short __cfwr_p1) {
        String __cfwr_temp36 = "world48";
        return null;
        return null;
    }
    public double __cfwr_util267(Boolean __cfwr_p0, Integer __cfwr_p1, byte __cfwr_p2) {
        if (true || (false & 2.77f)) {
            Long __cfwr_node7 = null;
        }
        return 7.18;
    }
}
