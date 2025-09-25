import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3")
    @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2")
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionsfunc1() {
        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
}
